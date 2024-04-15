import { ProcessFormConfs } from "@/features/processes/components/organisms/ProcessFormOrg/ProcessFormConfs";
import { ProcessFormError } from "@/features/processes/components/organisms/ProcessFormOrg/ProcessFormError";
import { BorderBoxAtm } from "@/features/shared/components/atoms/BorderBoxAtm";
import { HeaderAtm } from "@/features/shared/components/atoms/HeaderAtm";
import { components } from "@/services/backend/endpoints";
import { Tab } from "@headlessui/react";

interface Props {
  card: components["schemas"]["CardMod"];
  onSuccessSubmit?: () => void;
}

export function ProcessFormOrg({ card, onSuccessSubmit }: Props) {
  return (
    <BorderBoxAtm className="w-72 space-y-4 bg-black p-4">
      <Tab.Group>
        <Tab.List className="space-x-3">
          {card.status.type === "Errored" && (
            <>
              <Tab className="outline-none">
                {({ selected }) => (
                  <HeaderAtm
                    name="Error"
                    className={selected ? "select-none" : "opacity-50"}
                  />
                )}
              </Tab>
              <span className="border-x" />
            </>
          )}
          <Tab className="outline-none">
            {({ selected }) => (
              <HeaderAtm
                name="Configurations"
                className={selected ? "select-none" : "opacity-50"}
              />
            )}
          </Tab>
        </Tab.List>
        <Tab.Panels>
          {card.status.type === "Errored" && (
            <Tab.Panel>
              <ProcessFormError
                error={card.status.error}
                imageId={card.image_id}
                onSuccessSubmit={onSuccessSubmit}
              />
            </Tab.Panel>
          )}
          <Tab.Panel>
            <ProcessFormConfs card={card} onSuccessSubmit={onSuccessSubmit} />
          </Tab.Panel>
        </Tab.Panels>
      </Tab.Group>
    </BorderBoxAtm>
  );
}
