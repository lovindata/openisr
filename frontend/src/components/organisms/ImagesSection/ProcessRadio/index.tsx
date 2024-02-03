import { HorizontalRadio } from "../../../molecules/HorizontalRadio";
import { ProcessOptions } from "./ProcessOptions";

interface Props {
  value: ProcessOptions;
  setValue: (value: ProcessOptions) => void;
}

export function ProcessRadio({ value, setValue }: Props) {
  const values = Object.values(ProcessOptions);
  return (
    <HorizontalRadio
      possibleValues={values}
      value={value}
      setValue={setValue}
      className="w-72"
    />
  );
}
