import { ModalContext, ModalProvider } from ".";
import { useContext } from "react";

export function useModal() {
  const context = useContext(ModalContext);
  if (!context)
    throw Error(
      `Implementation error "${useModal.name}" must be used under "${ModalProvider.name}".`
    );
  return context;
}
